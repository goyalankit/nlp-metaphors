from nltk.corpus import verbnet

def GetVerbnetRestrictions(vnclass):
  role_restrictions = {}

  while True:
    for role in vnclass.findall('THEMROLES/THEMROLE'):
      restrictions = role.find('SELRESTRS')
      if restrictions:
        restriction_set = set()
        for restriction in restrictions.findall('SELRESTR'):
          predicate = restriction.attrib
          restriction_set.add((predicate['Value'], predicate['type']))

        total = (restrictions.get('logic', 'and'), list(restriction_set))
        role_restrictions[role.attrib['type']] = total

    if vnclass.tag == 'VNCLASS':
      break
    else:
      parent_class = vnclass.attrib['ID'].rsplit('-', 1)[0]
      vnclass = verbnet.vnclass(parent_class)

  return role_restrictions

vnclasses = verbnet.classids('drink')
v=verbnet.vnclass('39.1-2')
GetVerbnetRestrictions(v)
